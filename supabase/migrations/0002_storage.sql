-- Storage buckets + RLS so the browser can upload directly to its own folder.

insert into storage.buckets (id, name, public)
  values ('uploads', 'uploads', false)
  on conflict (id) do nothing;
insert into storage.buckets (id, name, public)
  values ('outputs', 'outputs', false)
  on conflict (id) do nothing;

-- Users may upload/read only under their own uid/ prefix in `uploads`.
create policy "uploads: own folder insert"
  on storage.objects for insert to authenticated
  with check (bucket_id = 'uploads' and (storage.foldername(name))[1] = auth.uid()::text);

create policy "uploads: own folder read"
  on storage.objects for select to authenticated
  using (bucket_id = 'uploads' and (storage.foldername(name))[1] = auth.uid()::text);

-- Outputs are written by the service-role worker; users read their own via the
-- API's signed download URLs (no direct browser policy needed). If you want
-- direct reads, add an analogous own-folder select policy on 'outputs'.
